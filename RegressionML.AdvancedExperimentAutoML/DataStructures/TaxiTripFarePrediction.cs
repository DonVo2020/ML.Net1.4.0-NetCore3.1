using Microsoft.ML.Data;

namespace RegressionML.AdvancedExperimentAutoML.DataStructures
{
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}