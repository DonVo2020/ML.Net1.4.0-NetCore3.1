using Microsoft.ML.Data;

namespace BinaryClassificationML.Beer
{
    public class BeerOrWinePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Beer;

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
