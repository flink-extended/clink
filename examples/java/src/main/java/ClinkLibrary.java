import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public class ClinkLibrary {

  public int ClinkInit(String remotePath, String localPath) {
    Pointer p_remote = new Memory((remotePath.length() + 1) * Native.WCHAR_SIZE);
    p_remote.setString(0, remotePath);
    Pointer p_local = new Memory((localPath.length() + 1) * Native.WCHAR_SIZE);
    p_local.setString(0, localPath);
    return Clibrary.INSTANTCE.FeatureOfflineInit(p_remote, p_local);
  }

  public String FeatureExtract(String input) {
    Pointer p_input = new Memory((input.length() + 1) * Native.WCHAR_SIZE);
    p_input.setString(0, input);
    PointerByReference ptrRef = new PointerByReference(Pointer.NULL);
    int res = Clibrary.INSTANTCE.FeatureExtractOffline(p_input, ptrRef);
    if (res != 0) {
      return null;
    }
    final Pointer p = ptrRef.getValue();
    // extract the null-terminated string from the Pointer
    final String val = p.getString(0);
    Clibrary.INSTANTCE.FeatureOfflineCleanUp(p);
    return val;
  }
}
