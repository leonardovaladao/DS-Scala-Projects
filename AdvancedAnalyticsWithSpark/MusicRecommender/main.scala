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
