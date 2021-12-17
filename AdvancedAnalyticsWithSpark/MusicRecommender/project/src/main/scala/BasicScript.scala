package MusicRecommenderModel

import org.apache.spark.sql.SparkSession

object BasicScript extends App {
  def main(): Unit = {
    val spark =
      SparkSession
        .builder
        .appName("Hello Spark App")
        .config("spark.master", "local")
        .config("spark.eventLog.enabled", false)
        .getOrCreate()

    val sparkConf = spark.conf
    sparkConf.set("spark.driver.memory", "10g")
    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._
    import org.apache.spark.sql.functions._



    println("Initializing...")

    // Data gathered from: https://storage.googleapis.com/aas-data-sets/profiledata_06-May-2005.tar.gz
    println("Loading user_artist_data...")
    val rawUserArtistData = spark.read.textFile("./src/user/ds/user_artist_data.txt")
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
    val rawArtistData = spark.read.textFile("./src/user/ds/artist_data.txt")
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
    val rawArtistAlias = spark.read.textFile("./src/user/ds/artist_alias.txt")
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
    val trainDataAll = buildCounts(rawUserArtistData, bArtistAlias)
    println("Cache train data")
    trainDataAll.cache()

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
      fit(trainDataAll)

    recommenderModel.userFactors.show(1, truncate = false)

    println("\n====================================================\n")
    println("Checking some user's history...")

    val userID = 2093760
    val existingArtistIDs = trainDataAll.filter($"user"===userID).select("artist").as[Int].collect()
    artistByID.filter($"id" isin (existingArtistIDs:_*)).show()

    println("Define function to make recommendations...")
    def makeRecommendations(model: ALSModel, userID: Int, howMany: Int) : DataFrame = {
      val toRecommend = recommenderModel.itemFactors.select($"id".as("artist")).withColumn("user", lit(userID))
      recommenderModel.transform(toRecommend).select("artist", "prediction").orderBy($"prediction".desc).limit(howMany)
    }

    println("See recomendations to user selected...")
    spark.conf.set( "spark.sql.crossJoin.enabled" , "true" )
    val topRecommendations = makeRecommendations(recommenderModel, userID, 5)
    topRecommendations.show()
    println("See recommendations by artist name...")
    val recommendedArtistIDs = topRecommendations.select("artist").as[Int].collect()
    artistByID.filter($"id" isin (recommendedArtistIDs:_*)).show()

    println("Results seems fine, but could definitely be better....")

    println("\n====================================================\n")
    /*
    import org.apache.spark.mllib.recommendation._
    import org.apache.spark.rdd.RDD
    */
    println("Import libraries for AUC...")
    import scala.collection.mutable.ArrayBuffer

    println("Define area under curve (AUC) function...")
    // This function was taken from
    def areaUnderCurve(
                        positiveData: DataFrame,
                        bAllArtistIDs: Broadcast[Array[Int]],
                        predictFunction: (DataFrame => DataFrame)): Double = {

      // What this actually computes is AUC, per user. The result is actually something
      // that might be called "mean AUC".

      // Take held-out data as the "positive".
      // Make predictions for each of them, including a numeric score
      val positivePredictions = predictFunction(positiveData.select("user", "artist")).
        withColumnRenamed("prediction", "positivePrediction")

      // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
      // small AUC problems, and it would be inefficient, when a direct computation is available.

      // Create a set of "negative" products for each user. These are randomly chosen
      // from among all of the other artists, excluding those that are "positive" for the user.
      val negativeData = positiveData.select("user", "artist").as[(Int,Int)].
        groupByKey { case (user, _) => user }.
        flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
          val random = new Random()
          val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
          val negative = new ArrayBuffer[Int]()
          val allArtistIDs = bAllArtistIDs.value
          var i = 0
          // Make at most one pass over all artists to avoid an infinite loop.
          // Also stop when number of negative equals positive set size
          while (i < allArtistIDs.length && negative.size < posItemIDSet.size) {
            val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))
            // Only add new distinct IDs
            if (!posItemIDSet.contains(artistID)) {
              negative += artistID
            }
            i += 1
          }
          // Return the set with user ID added back
          negative.map(artistID => (userID, artistID))
        }.toDF("user", "artist")

      // Make predictions on the rest:
      val negativePredictions = predictFunction(negativeData).
        withColumnRenamed("prediction", "negativePrediction")

      // Join positive predictions to negative predictions by user, only.
      // This will result in a row for every possible pairing of positive and negative
      // predictions within each user.
      val joinedPredictions = positivePredictions.join(negativePredictions, "user").
        select("user", "positivePrediction", "negativePrediction").cache()

      // Count the number of pairs per user
      val allCounts = joinedPredictions.
        groupBy("user").agg(count(lit("1")).as("total")).
        select("user", "total")
      // Count the number of correctly ordered pairs per user
      val correctCounts = joinedPredictions.
        filter($"positivePrediction" > $"negativePrediction").
        groupBy("user").agg(count("user").as("correct")).
        select("user", "correct")

      // Combine these, compute their ratio, and average over all users
      val meanAUC = allCounts.join(correctCounts, Seq("user"), "left_outer").
        select($"user", (coalesce($"correct", lit(0)) / $"total").as("auc")).
        agg(mean("auc")).
        as[Double].first()

      joinedPredictions.unpersist()

      meanAUC
    }

    println("Split trainData to get test data...")
    val allData = buildCounts(rawUserArtistData, bArtistAlias)
    val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    cvData.cache()

    println("Collect all artist data...")
    val allArtistIDs = allData.select("artist").as[Int].distinct().collect()
    val bAllArtistIDs = spark.sparkContext.broadcast(allArtistIDs)

    println("Re-evaluate model with new trainData...")
    val recommenderModel2 = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      setRank(10).
      setRegParam(0.01).
      setAlpha(1.0).
      setMaxIter(5).
      setUserCol("user").setItemCol("artist").
      setRatingCol("count").setPredictionCol("prediction").
      fit(trainData)

    println("Evaluate model through AUC, tested on testData...")
    val aucModel = areaUnderCurve(cvData, bAllArtistIDs, recommenderModel2.transform)
    println("The above model could gather "+aucModel+" of accuracy.") // 0.9005

    println("Let's test the AUC with a simpler model: Recommend the most played artists for every user...")
    def predictMostListened(train: DataFrame)(allData: DataFrame) = {
      val listenCounts = train.groupBy("artist").agg(sum("count").as("prediction")).select("artist", "prediction")
      allData.join(listenCounts, Seq("artist"), "left_outer").select("user", "artist", "prediction")
    }
    val aucMostListened = areaUnderCurve(cvData, bAllArtistIDs, predictMostListened(trainData))
    println("That's "+aucMostListened+" of accuracy. Suddenly, our model doesn't look so impressive...")
    println("Clearly our model need some tuning...")

    println("Now training with the best parameters...")

    val recommenderModel3 = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      setRank(30).
      setRegParam(4).
      setAlpha(40).
      setMaxIter(5).
      setUserCol("user").setItemCol("artist").
      setRatingCol("count").setPredictionCol("prediction").
      fit(trainData)

    val aucModelBest = areaUnderCurve(cvData, bAllArtistIDs, recommenderModel3.transform)
    println("With a very better model, we could reach AUC with "+aucModelBest) // 0.9142

    spark.stop()

  }
  main()
}
