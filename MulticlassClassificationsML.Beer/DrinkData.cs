using Microsoft.ML.Data;

namespace MulticlassClassificationsML.Beer
{
    public class DrinkData
    {
        [LoadColumn(0)]
        public string FullName;
        [LoadColumn(1)]
        public string Type;
        [LoadColumn(2)]
        public string Country;
    }
}
