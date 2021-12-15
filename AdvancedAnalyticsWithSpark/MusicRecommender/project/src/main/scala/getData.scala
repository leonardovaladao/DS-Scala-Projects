package MusicRecommenderModel

import MusicRecommenderModel.getSpark._

object getData {
  val spark = getSpark.spark()

  import spark.implicits._
  import org.apache.spark.sql.functions._

  println("1")
  // Loading user_artist_data...
  val rawUserArtistData = spark.read.textFile("./src/user/ds/user_artist_data.txt")
  // Loading artist's names...
  val rawArtistData = spark.read.textFile("./src/user/ds/artist_data.txt")
  // Loading artist alias file...
  val rawArtistAlias = spark.read.textFile("./src/user/ds/artist_alias.txt")

  println("2")
  // Creating userArtistDF representing artist's DataFrame
  val userArtistDF = rawUserArtistData.map { line =>
    val Array(user, artist, _*) = line.split(' ')
    (user.toInt, artist.toInt)
  }.toDF("user", "artist")

  println("3")
  // Creating artistByID DF, representing each artist associated with an ID...
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

  println("4")
  // Cleaning artist alias file, so we can have one ID per artist...
  val artistAlias = rawArtistAlias.flatMap { line =>
    val Array(artist, alias) = line.split('\t')
    if (artist.isEmpty) {
      None
    } else {
      Some((artist.toInt, alias.toInt))
    }
  }.collect().toMap

  println("5")

  // Import Spark SQL and Broadcast...
  import org.apache.spark.sql._
  import org.apache.spark.broadcast._

  println("6")
  // Get unique name for each artist, and gather with count for each user...
  def buildCounts(rawUserArtistData: Dataset[String], bArtistAlias: Broadcast[Map[Int, Int]]): DataFrame = {
    rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      (userID, finalArtistID, count)
    }.toDF("user", "artist", "count")
  }

  println("7")
  // Set broadcast...
  val bArtistAlias = spark.sparkContext.broadcast(artistAlias)

  spark.stop()
}
