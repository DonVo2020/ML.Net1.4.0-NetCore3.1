using Microsoft.ML.Data;

namespace RegressionML.BikeSharingDemand.DataStructures
{
    public class DemandPrediction
    {
        [ColumnName("Score")]
        public float PredictedCount;
    }
}
