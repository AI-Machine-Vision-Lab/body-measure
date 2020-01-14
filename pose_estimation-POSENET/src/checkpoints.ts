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

import {ConvolutionDefinition, mobileNetArchitectures} from './mobilenet'

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/tfjs-models/weights/posenet/';

export type Checkpoint = {
  url: string,
  architecture: ConvolutionDefinition[]
}

export const checkpoints: {[multiplier: number]: Checkpoint} = {
  1.01: {
    url: GOOGLE_CLOUD_STORAGE_DIR + 'mobilenet_v1_101/',
    architecture: mobileNetArchitectures[100]
  },
  1.0: {
    url: GOOGLE_CLOUD_STORAGE_DIR + 'mobilenet_v1_100/',
    architecture: mobileNetArchitectures[100]
  },
  0.75: {
    url: GOOGLE_CLOUD_STORAGE_DIR + 'mobilenet_v1_075/',
    architecture: mobileNetArchitectures[75]
  },
  0.5: {
    url: GOOGLE_CLOUD_STORAGE_DIR + 'mobilenet_v1_050/',
    architecture: mobileNetArchitectures[50]
  }
}
