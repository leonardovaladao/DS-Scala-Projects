package MusicRecommenderModel

object getSpark {
  println("Importing SparkSession.")

  import org.apache.spark.SparkConf
  import org.apache.spark.sql.SparkSession

  def spark(): SparkSession = {
    val conf = new SparkConf().setMaster("local[*]").set("spark.driver.memory", "12g")

    val spark = SparkSession
      .builder
      .appName("MusicRecommenderModel")
      .config(conf)
      .config("spark.eventLog.enabled", false)
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    /*println("How much GB of RAM do you want to use? Answer as an integer:")
    val ram = scala.io.StdIn.readInt()
    spark.conf.set("spark.executor.memory", ram.toString + "g")
    println("Using " + ram + "GB of RAM.")

    println("How much GB of the driver do you want to use? Answer as an integer:")
    val driver = scala.io.StdIn.readInt()
    spark.conf.set("spark.driver.memory", driver.toString+"g")
    println("Using "+driver+"GB of the driver.")*/

    //spark.conf.set("spark.executor.memory", "12g")
    //spark.conf.set("spark.driver.memory", "12g")
    spark.sparkContext.getConf.getAll.foreach(println)
    spark
  }
}