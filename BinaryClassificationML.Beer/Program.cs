﻿using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace BinaryClassificationML.Beer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Binary Classification");

            // Define context
            var mlContext = new MLContext();

            // Load training data
            var trainingDataView = mlContext.Data.LoadFromTextFile<BeerOrWineData>(
                @"C:\DEVELOPMENT\Machine Learning Projects\DonVo.ML110\BinaryClassificationML.Beer\datasets\problem1_train.csv",
                hasHeader: true,
                separatorChar: ',');
            // Define features
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "FullName");

            // Use Binary classification
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Beer", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train the model based on training data
            var watch = Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();

            Console.WriteLine($"Trained the model in: {watch.ElapsedMilliseconds / 1000} seconds.");

            // Use model for predictions
            List<BeerOrWineData> drinks = new List<BeerOrWineData>
            {
                new BeerOrWineData { FullName = "Castle Stout" },
                new BeerOrWineData { FullName = "Folkes Röda IPA"},
                new BeerOrWineData { FullName = "Fryken Havre Ale"},
                new BeerOrWineData { FullName = "Barolo Gramolere"},
                new BeerOrWineData { FullName = "Château de Lavison"},
                new BeerOrWineData { FullName = "Korlat Cabernet Sauvignon"}
            };

            var predFunction = mlContext.Model.CreatePredictionEngine<BeerOrWineData, BeerOrWinePrediction>(trainedModel);

            foreach (var drink in drinks)
            {
                var prediction = predFunction.Predict(drink);

                var isBeer = prediction.Beer ? "Beer" : "Wine";

                Console.WriteLine($"{drink.FullName} is {isBeer}");
            }

            // Evaluate the model
            var testDataView = mlContext.Data.LoadFromTextFile<BeerOrWineData>(
                @"C:\DEVELOPMENT\Machine Learning Projects\DonVo.ML110\BinaryClassificationML.Beer\datasets\problem1_validate.csv",
                hasHeader: true,
                separatorChar: ',');
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Beer", "Score");

            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");

            // Cross validation
            var fullDataView = mlContext.Data.LoadFromTextFile<BeerOrWineData>(
                @"C:\DEVELOPMENT\Machine Learning Projects\DonVo.ML110\BinaryClassificationML.Beer\datasets\problem1.csv",
                hasHeader: true,
                separatorChar: ',');
            var cvResults = mlContext.BinaryClassification.CrossValidate(fullDataView, trainingPipeline, numberOfFolds: 3, labelColumnName: "Beer");
            Console.WriteLine($"Avg Accuracy is: {cvResults.Select(r => r.Metrics.Accuracy).Average():P2}");

            Console.WriteLine();
        }
    }
}
