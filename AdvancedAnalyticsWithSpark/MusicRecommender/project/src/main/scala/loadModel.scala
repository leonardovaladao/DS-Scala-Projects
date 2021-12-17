package MusicRecommenderModel

import MusicRecommenderModel.getData.loadData

object loadModel extends App{
  val spark = loadData

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
}
