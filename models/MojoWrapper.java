import java.io.*;
import com.google.gson.*;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;
import java.lang.Integer;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.prediction.*;
import hex.genmodel.MojoModel;

public class MojoWrapper {
  public static void main(String[] args) throws Exception {
    EasyPredictModelWrapper model = 
      new EasyPredictModelWrapper(MojoModel.load(
          "GBM_model_python_1524899144108_1.zip"));

    // args input to HashMap
    String json = args[0];
    Gson gson = new Gson(); 
    Map<String,String> map = new HashMap<String,String>();
    map = (Map<String,String>) gson.fromJson(json, map.getClass());

    RowData row = new RowData();
    row.putAll(map);

    MultinomialModelPrediction p = model.predictMultinomial(row);
    String json_str = "{\"precition\":\""+ p.label + "\"," +
        " \"probability\":\"" + p.classProbabilities[p.labelIndex] + "\"}";
    System.out.println(json_str);
  }
}
