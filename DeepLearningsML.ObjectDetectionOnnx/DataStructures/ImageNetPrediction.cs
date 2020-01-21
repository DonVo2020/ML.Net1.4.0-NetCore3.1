using Microsoft.ML.Data;

namespace DeepLearningsML.ObjectDetectionOnnx
{
    public class ImageNetPrediction
    {
        [ColumnName(OnnxModelScorer.TinyYoloModelSettings.ModelOutput)]
        public float[] PredictedLabels;
    }
}
