using Microsoft.ML.Data;

namespace MulticlassClassificationsML.WordsClassification
{
    public class WordModelPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Category { get; set; }
    }
}
