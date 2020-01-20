using Microsoft.ML.Data;
using System;

namespace AnomalyDetectionsML.PowerMeterReadings.DataStructures
{
    class MeterData
    {
        [LoadColumn(0)]
        public string name { get; set; }
        [LoadColumn(1)]
        public DateTime time { get; set; }
        [LoadColumn(2)]
        public float ConsumptionDiffNormalized { get; set; }
    }
}
