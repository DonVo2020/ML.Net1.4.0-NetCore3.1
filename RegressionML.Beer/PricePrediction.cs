using Microsoft.ML.Data;

namespace RegressionML.Beer
{
    public class PricePrediction
    {
        [ColumnName("Score")]
        public float Price;
    }
}
