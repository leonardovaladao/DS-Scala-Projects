package MusicRecommenderModel

import MusicRecommenderModel.getSpark._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object getData {
  def loadData: SparkSession = {
    val spark = getSpark.spark()


    val rawUserArtistData = spark.read.textFile("./src/user/ds/user_artist_data.txt")
    val rawArtistAlias = spark.read.textFile("./src/user/ds/artist_alias.txt")
    val rawArtistData = spark.read.textFile("./src/user/ds/artist_data.txt")

    import spark.implicits._
    import org.apache.spark.sql.functions._


    println("Creating userArtistDF representing artist's DataFrame...")
    val userArtistDF = rawUserArtistData.map { line =>
      val Array(user, artist, _*) = line.split(' ')
      (user.toInt, artist.toInt)
    }.toDF("user", "artist")

    println("Loading artist's names...")
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

    println("Cleaning previous file with a map function to find two IDs for same artists...")
    val artistAlias = rawArtistAlias.flatMap { line =>
      val Array(artist, alias) = line.split('\t')
      if (artist.isEmpty) {
        None
      } else {
        Some((artist.toInt, alias.toInt))
      }
    }.collect().toMap

    println("Import Spark SQL and Spark Broadcast")

    import org.apache.spark.sql._
    import org.apache.spark.broadcast._

    println("Get unique name for each artist, and gather with count for each user...")

    def buildCounts(rawUserArtistData: Dataset[String], bArtistAlias: Broadcast[Map[Int, Int]]): DataFrame = {
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

    spark
  }
}