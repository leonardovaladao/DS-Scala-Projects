println("Initializing...")

// Data gathered from: https://storage.googleapis.com/aas-data-sets/profiledata_06-May-2005.tar.gz
println("Loading user_artist_data...")
val rawUserArtistData = spark.read.textFile("./user/ds/user_artist_data.txt")
println("Show first five rows...")
println("Data format: \n userID | artistID | play count")
rawUserArtistData.take(5).foreach(println)

println("Creating userArtistDF representing artist's DataFrame...")
val userArtistDF = rawUserArtistData.map { line =>
    val Array(user, artist, _*) = line.split(' ')
    (user.toInt, artist.toInt)
}.toDF("user", "artist")
println("Show first five rows of userArtistDF...")
userArtistDF.show(5)

println("Show results of min and max for each value...")
userArtistDF.agg(min("user"), max("user"), min("artist"), max("artist")).show()

println("Loading artist's names...")
val rawArtistData = spark.read.textFile("./user/ds/artist_data.txt")
val artistByID = rawArtistData.flatMap { line =>
    val (id, name) = line.span(_ != '\t')
    if (name.isEmpty) {
        None
    } else {
        try {
            Some((id.toInt, name.trim))
        } catch {
            case _: NumberFormatException => None
        }
    } 
}.toDF("id", "name")

println("Loading artist alias file...")
val rawArtistAlias = spark.read.textFile("./user/ds/artist_alias.txt")
println("Cleaning previous file with a map function to find two IDs for same artists...")
val artistAlias = rawArtistAlias.flatMap { line =>
    val Array(artist, alias) = line.split('\t')
    if (artist.isEmpty) {
        None
    } else {
        Some((artist.toInt, alias.toInt))
    }
}.collect().toMap

println("Show first line in artistAlias...")
artistAlias.take(2).tail
println("Show name of artist...")
artistByID.filter($"id" isin (2012757,4569)).show()

println("\n====================================================\n")
println("Import Spark SQL and Spark Broadcast")

import org.apache.spark.sql._ 
import org.apache.spark.broadcast._ 

println("Get unique name for each artist, and gather with count for each user...")
def buildCounts (rawUserArtistData: Dataset[String], bArtistAlias: Broadcast[Map[Int, Int]]): DataFrame = {
    rawUserArtistData.map { line => 
        val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
        val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
        (userID, finalArtistID, count)
    }.toDF("user", "artist", "count")
}

println("Set broadcast...")
val bArtistAlias = spark.sparkContext.broadcast(artistAlias)
println("Define train data")
val trainData = buildCounts(rawUserArtistData, bArtistAlias)
println("Cache train data")
trainData.cache()

println("\n====================================================\nHere we will begin our ML model!")
import org.apache.spark.ml.recommendation._
import scala.util.Random

val recommenderModel = new ALS().
    setSeed(Random.nextLong()).
    setImplicitPrefs(true).
    setRank(10).
    setRegParam(0.01).
    setAlpha(1.0).
    setMaxIter(5).
    setUserCol("user").
    setItemCol("user").
    setItemCol("artist").
    setRatingCol("count").
    setPredictionCol("prediction").
    fit(trainData)

recommenderModel.userFactors.show(1, truncate = false)

println("\n====================================================\n")
println("Checking some user's history...")

val userID = 2093760
val existingArtistIDs = trainData.filter($"user"===userID).select("artist").as[Int].collect()
artistByID.filter($"id" isin (existingArtistIDs:_*)).show()

println("Define function to make recommendations...")
def makeRecommendations(model: ALSModel, userID: Int, howMany: Int) : DataFrame = {
    val toRecommend = recommenderModel.itemFactors.select($"id".as("artist")).withColumn("user", lit(userID))
    recommenderModel.transform(toRecommend).select("artist", "prediction").orderBy($"prediction".desc).limit(howMany)
}

println("See recomendations to user selected...")
val topRecommendations = makeRecommendations(recommenderModel, userID, 5)
topRecommendations.show()
println("See recommendations by artist name...")
val recommendedArtistIDs = topRecommendations.select("artist").as[Int].collect()
artistByID.filter($"id" isin (recommendedArtistIDs:_*)).show()

println("Results seems fine, but could definitely be better....")

println("\n====================================================\n")

println("Import libraries for AUC...")
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD

println("Define area under curve (AUC) function...")
// This function was taken from https://github.com/mauropelucchi/machine-learning-course/blob/master/collaborative-filtering/recommending_music.scala
def areaUnderCurve(
      positiveData: RDD[Rating],
      bAllItemIDs: Broadcast[Array[Int]],
      predictFunction: (RDD[(Int,Int)] => RDD[Rating])) = {
    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // Take held-out data as the "positive", and map to tuples
    val positiveUserProducts = positiveData.map(r => (r.user, r.product))
    // Make predictions for each of them, including a numeric score, and gather by user
    val positivePredictions = predictFunction(positiveUserProducts).groupBy(_.user)

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other items, excluding those that are "positive" for the user.
    val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions {
      // mapPartitions operates on many (user,positive-items) pairs at once
      userIDAndPosItemIDs => {
        // Init an RNG and the item IDs set once for partition
        val random = new Random()
        val allItemIDs = bAllItemIDs.value
        userIDAndPosItemIDs.map { case (userID, posItemIDs) =>
          val posItemIDSet = posItemIDs.toSet
          val negative = new ArrayBuffer[Int]()
          var i = 0
          // Keep about as many negative examples per user as positive.
          // Duplicates are OK
          while (i < allItemIDs.size && negative.size < posItemIDSet.size) {
            val itemID = allItemIDs(random.nextInt(allItemIDs.size))
            if (!posItemIDSet.contains(itemID)) {
              negative += itemID
            }
            i += 1
          }
          // Result is a collection of (user,negative-item) tuples
          negative.map(itemID => (userID, itemID))
        }
      }
    }.flatMap(t => t)
    // flatMap breaks the collections above down into one big set of tuples

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeUserProducts).groupBy(_.user)

    // Join positive and negative by user
    positivePredictions.join(negativePredictions).values.map {
      case (positiveRatings, negativeRatings) =>
        // AUC may be viewed as the probability that a random positive item scores
        // higher than a random negative one. Here the proportion of all positive-negative
        // pairs that are correctly ranked is computed. The result is equal to the AUC metric.
        var correct = 0L
        var total = 0L
        // For each pairing,
        for (positive <- positiveRatings;
             negative <- negativeRatings) {
          // Count the correctly-ranked pairs
          if (positive.rating > negative.rating) {
            correct += 1
          }
          total += 1
        }
        // Return AUC: fraction of pairs ranked correctly
        correct.toDouble / total
    }.mean() // Return mean AUC over users
  }

println("Split trainData to get test data...")
val allData = buildCounts(rawUserArtistData, bArtistAlias)
val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
trainData.cache()
cvData.cache()

val allArtistIDs = allData.select("artist").as[Int].distinct().collect()
val bAllArtistIDs = spark.sparkContext.broadcast(allArtistIDs) 

import org.apache.spark.ml.recommendation._

val model = new ALS().
  setSeed(Random.nextLong()).
  setImplicitPrefs(true).
  setRank(10).setRegParam(0.01).setAlpha(1.0).setMaxIter(5).
  setUserCol("user").setItemCol("artist").
  setRatingCol("count").setPredictionCol("prediction").
  fit(trainData)