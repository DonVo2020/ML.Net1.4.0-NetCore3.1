using Microsoft.ML.Data;

namespace RegressionML.Wine
{
    public class WinePrediction
    {
        [ColumnName("Score")]
        public float Quality;
    }
}
