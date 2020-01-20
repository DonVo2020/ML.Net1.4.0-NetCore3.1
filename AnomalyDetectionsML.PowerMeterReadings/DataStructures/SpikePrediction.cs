using Microsoft.ML.Data;

namespace AnomalyDetectionsML.PowerMeterReadings.DataStructures
{
    class SpikePrediction
    {
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
