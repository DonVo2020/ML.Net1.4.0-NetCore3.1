using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace Predictions.NGrams.Text
{

    /// <summary>
    /// Wrapper for ML.Net Text Normalizer
    /// </summary>
    public class TextNormalizer : IDisposable
    {
        private MLContext _mlContext;
        private List<TextData> _emptySamplesList;
        private IDataView _emptyDataView;
        
        // text normalization
        private TextNormalizingEstimator _normTextPipeline;
        private TextNormalizingTransformer _normTextTransformer;
        private PredictionEngine<TextData, TransformedTextData> _predictionEngine;
        
        public TextNormalizer()
        {
            InitializeTextNormalizer();            
        }

        public TextNormalizer(TextNormalizingEstimator.CaseMode caseMode = TextNormalizingEstimator.CaseMode.Lower,
            bool keepDiacritics = false,
            bool keepPuncuations = false,
            bool keepNumbers = false)
        {
            InitializeTextNormalizer(caseMode, keepDiacritics, keepPuncuations, keepNumbers);
        }

        private void InitializeTextNormalizer(TextNormalizingEstimator.CaseMode caseMode = TextNormalizingEstimator.CaseMode.Lower,
            bool keepDiacritics = false,
            bool keepPuncuations = false,
            bool keepNumbers = false)
        {
            _mlContext = new MLContext();
            _emptySamplesList = new List<TextData>();
            _emptyDataView = _mlContext.Data.LoadFromEnumerable(_emptySamplesList);

            // text normalizer
            _normTextPipeline = _mlContext.Transforms.Text.NormalizeText("NormalizedText", "Text",
                caseMode,
                keepDiacritics: keepDiacritics,
                keepPunctuations: keepPuncuations,
                keepNumbers: keepNumbers);
            _normTextTransformer = _normTextPipeline.Fit(_emptyDataView);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(_normTextTransformer);
        }



        /// <summary>
        /// Normalize input text.
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        public string NormalizeText(string text)
        {                               
            var data = new TextData() { Text = text };
            var prediction = _predictionEngine.Predict(data);
            return prediction.NormalizedText;
        }


        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData : TextData
        {
            public string NormalizedText { get; set; }
        }


        public void Dispose()
        {
            _predictionEngine?.Dispose();            
        }
    }    
}
