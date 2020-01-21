using Microsoft.ML.Data;

namespace RegressionML.Beer
{
    public class PriceData
    {
        [LoadColumn(0)]
        public string FullName;
        [LoadColumn(1)]
        public float Price;
        [LoadColumn(2)]
        public float Volume;
        [LoadColumn(3)]
        public string Type;
        [LoadColumn(4)]
        public string Country;
    }
}
