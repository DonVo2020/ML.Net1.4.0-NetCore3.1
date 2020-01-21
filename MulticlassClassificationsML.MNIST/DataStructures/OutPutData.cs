using Microsoft.ML.Data;

namespace MulticlassClassificationsML.MNIST.DataStructures
{
    class OutPutData
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}
