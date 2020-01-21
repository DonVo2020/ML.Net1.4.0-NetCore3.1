using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace Predictions.NGrams.Text
{
    public class StopWordsRemover : IDisposable
    {

        private MLContext _mlContext;
        private List<TextData> _emptySamplesList;
        private IDataView _emptyDataView;
        
        // stop words
        private Microsoft.ML.Data.EstimatorChain<StopWordsRemovingTransformer> _stopWordsTextPipeline;
        private Microsoft.ML.Data.TransformerChain<StopWordsRemovingTransformer> _textPipeline;
        private PredictionEngine<TextData, TransformedTextData> _predictionEngine;

        private removeStopWordswMode _mode;

        private enum removeStopWordswMode
        {
            Custom = 0,
            Default = 1
        }

        /// <summary>
        /// constructor for default stopwords
        /// </summary>
        public StopWordsRemover()
        {
            InitializeStopWordsRemover();    
        }

        /// <summary>
        /// constructor for custom stop words
        /// </summary>
        /// <param name="stopWords"></param>
        public StopWordsRemover(string[] stopWords)
        {
            InitializeStopWordsRemover(stopWords);
        }


        /// <summary>
        /// Initialize (or reinitialize) the stopwords remover.  Also called by the constructor.
        /// </summary>
        /// <param name="CustomStopWords"></param>
        public void InitializeStopWordsRemover(string[] CustomStopWords = null)
        {
            _mlContext = new MLContext();
            _emptySamplesList = new List<TextData>();
            _emptyDataView = _mlContext.Data.LoadFromEnumerable(_emptySamplesList);

            // stop words
            _stopWordsTextPipeline = _mlContext.Transforms.Text.TokenizeIntoWords("Words", "Text")
                .Append(_mlContext.Transforms.Text.RemoveDefaultStopWords("WordsWithoutStopWords", "Words", language: StopWordsRemovingEstimator.Language.English));

            if (CustomStopWords == null)
            {
                _textPipeline = _stopWordsTextPipeline.Fit(_emptyDataView);
                _mode = removeStopWordswMode.Default;
            }
            else
            {
                //var textPipeline = _mlContext.Transforms.Text.TokenizeIntoWords("Words",
                //"Text")
                //.Append(_mlContext.Transforms.Text.RemoveStopWords(
                //"WordsWithoutStopWords", "Words", stopwords:
                //CustomStopWords));
                _mode = removeStopWordswMode.Custom;
            }
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(_textPipeline);
        }


        /// <summary>
        /// remove stop words from input string
        /// </summary>
        /// <param name="inputString"></param>
        /// <returns></returns>
        public string RemoveStopWords(string inputString)
        {
            var data = new TextData() { Text = inputString };
            var prediction = _predictionEngine.Predict(data);

            string retString;
            if (prediction.WordsWithoutStopWords != null)
            {
                retString = string.Join(" ", prediction.WordsWithoutStopWords);
            }
            else
            {
                retString = inputString;
            }   
            return retString;
        }

        public void ShowCurrentMode()
        {
            Console.WriteLine($"Stopwords remover Mode: {_mode.ToString()}");
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData : TextData
        {
            public string[] WordsWithoutStopWords { get; set; }
        }

        public void Dispose()
        {
            _predictionEngine?.Dispose();
        }
    }
}
