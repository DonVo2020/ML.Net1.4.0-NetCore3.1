using Microsoft.ML.Data;

namespace RegressionML.AutoML.DataStructures
{
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}