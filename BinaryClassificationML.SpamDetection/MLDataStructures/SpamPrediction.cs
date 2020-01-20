using Microsoft.ML.Data;

namespace BinaryClassificationML.SpamDetection.MLDataStructures
{
    class SpamPrediction
    {
        [ColumnName("PredictedLabel")]
        public string isSpam { get; set; }
    }
}
