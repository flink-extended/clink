
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public interface Clibrary extends Library {
  // 加载链接库
  String dllPath =
      "libclink.dylib";
  Clibrary INSTANTCE = Native.loadLibrary(dllPath, Clibrary.class);

  //初始化
  int FeatureOfflineInit(Pointer remote_url, Pointer local_path);
  
  // 此方法为链接库中的方法
  int FeatureExtractOffline(Pointer input, PointerByReference val);

  void FeatureOfflineCleanUp(Pointer p);
}
