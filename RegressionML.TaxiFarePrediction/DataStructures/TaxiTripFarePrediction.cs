using Microsoft.ML.Data;

namespace RegressionML.TaxiFarePrediction.DataStructures
{
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}