/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.clink.example;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.util.Properties;

/** Simple example of native library declaration and usage. */
public class Main {

    public interface ClinkKernels extends Library {
        ClinkKernels INSTANCE = Native.load("clink_kernels_jna", ClinkKernels.class);

        double SquareAdd(double x, double y);

        double Square(double x);
    }

    public static void main(String[] args) {
        System.out.println("Square result is " + ClinkKernels.INSTANCE.Square(3.0));
        System.out.println("SquareAdd result is " + ClinkKernels.INSTANCE.SquareAdd(1.0, 3.0));
    }
}
