/*
 * Copyright 2021 The Clink Authors
 *
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

package org.flinkextended.clink.util;

import org.apache.flink.table.shaded.org.reflections.Reflections;
import org.apache.flink.table.shaded.org.reflections.scanners.SubTypesScanner;

import junit.framework.JUnit4TestAdapter;
import junit.framework.TestSuite;
import org.junit.runner.RunWith;

/** Run all tests in org.clink. */
@RunWith(org.junit.runners.AllTests.class)
public class AllTestsRunner {
    public static TestSuite suite() throws Exception {
        TestSuite suite = new TestSuite();
        Reflections reflections =
                new Reflections("org/flinkextended/clink", new SubTypesScanner(false));
        reflections.getSubTypesOf(Object.class).stream()
                .filter(clazz -> clazz.getName().endsWith("Test"))
                .map(JUnit4TestAdapter::new)
                .forEach(suite::addTest);
        return suite;
    }
}
