val evaluations =
      for (rank     <- Seq(5,  30);
           regParam <- Seq(1.0, 0.0001);
           alpha    <- Seq(1.0, 40.0))
      yield {
        val model = new ALS().
          setSeed(Random.nextLong()).
          setImplicitPrefs(true).
          setRank(rank).setRegParam(regParam).
          setAlpha(alpha).setMaxIter(20).
          setUserCol("user").setItemCol("artist").
          setRatingCol("count").setPredictionCol("prediction").
          fit(trainData)

        val auc = areaUnderCurve(cvData, bAllArtistIDs, model.transform)

        model.userFactors.unpersist()
        model.itemFactors.unpersist()

        (auc, (rank, regParam, alpha))
      }