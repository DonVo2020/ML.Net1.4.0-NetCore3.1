using Microsoft.ML.Data;

namespace BinaryClassificationML.SpamDetection.MLDataStructures
{
    class SpamInput
    {
        [LoadColumn(0)]
        public string Label { get; set; }
        [LoadColumn(1)]
        public string Message { get; set; }
    }
}
