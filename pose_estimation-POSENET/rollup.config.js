/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licnses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import node from 'rollup-plugin-node-resolve';
import typescript from 'rollup-plugin-typescript2';

export default {
  input: 'src/index.ts',
  plugins: [
    typescript(),
    node()
  ],
  external: [
    '@tensorflow/tfjs'
  ],
  output: {
    banner: `// @tensorflow/tfjs-models Copyright ${(new Date).getFullYear()} Google`,
    file: 'dist/posenet.js',
    format: 'umd',
    name: 'posenet',
    globals: {
      '@tensorflow/tfjs': 'tf'
    }
  }
};
