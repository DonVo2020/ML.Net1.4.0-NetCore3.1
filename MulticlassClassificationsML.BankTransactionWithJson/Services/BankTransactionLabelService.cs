﻿using Microsoft.ML;
using MulticlassClassificationsML.BankTransactionWithJson.Models;
using System;
using System.IO;

namespace MulticlassClassificationsML.BankTransactionWithJson.Services
{
    public class BankTransactionLabelService
    {
        private readonly MLContext _mlContext;
        private PredictionEngine<TransactionData, BankTransactionPrediction> _predEngine;

        public BankTransactionLabelService()
        {
            _mlContext = new MLContext(seed: 0);
        }

        public void LoadModel(string modelPath)
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = _mlContext.Model.Load(stream, out var modelInputSchema);
            _predEngine = _mlContext.Model.CreatePredictionEngine<TransactionData, BankTransactionPrediction>(loadedModel);
        }

        public string PredictCategory(TransactionData transaction)
        {
            var prediction = new BankTransactionPrediction();
            _predEngine.Predict(transaction, ref prediction);
            if (prediction != null && !String.IsNullOrWhiteSpace(prediction.Category))
            {
                return prediction.Category;
            }
            return "Other";
        }
    }
}
