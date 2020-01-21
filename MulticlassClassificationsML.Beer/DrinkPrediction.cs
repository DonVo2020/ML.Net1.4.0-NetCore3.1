using Microsoft.ML.Data;

namespace MulticlassClassificationsML.Beer
{
    public class DrinkPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Type;

        [ColumnName("Score")]
        public float[] Scores;
    }
}
