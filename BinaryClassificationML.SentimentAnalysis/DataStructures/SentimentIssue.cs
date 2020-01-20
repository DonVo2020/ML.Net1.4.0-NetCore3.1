using Microsoft.ML.Data;

namespace BinaryClassificationML.SentimentAnalysis.DataStructures
{
    public class SentimentIssue
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
