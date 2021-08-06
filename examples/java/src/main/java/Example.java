import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Memory;
import com.sun.jna.ptr.PointerByReference;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
public class Example {
    public static void main(String... strings) {

    System.setProperty("jna.encoding", "UTF-8");
    // 调用
    String path = "./config";
    String remote = ""; //远程配置地址
    ClinkLibrary library = new ClinkLibrary();

    if(library.ClinkInit(remote, path)!=0){
      System.out.println("fail to init clink library");
      return;
    }
    String fileName = "../dataset/data.csv";
    File file = new File(fileName);
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new FileReader(file));
      String tempString = null;
      // 一次读入一行，直到读入null为文件结束
      while ((tempString = reader.readLine()) != null) {
        String out = library.FeatureExtract(tempString);
        System.out.println(out);
      }
      reader.close();
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      if (reader != null) {
        try {
          reader.close();
        } catch (IOException e1) {
        }
      }
    }
  }

}