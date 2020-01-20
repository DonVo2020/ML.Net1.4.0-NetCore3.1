using Microsoft.ML.Data;

namespace BinaryClassificationML.Beer
{
    public class BeerOrWineData
    {
        [LoadColumn(0)]
        public string FullName;
        [LoadColumn(1)]
        public bool Beer;
    }
}
