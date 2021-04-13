# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

from ..base import Parameter
from ..utilities import positive
from .base import Kernel


class DOT(Kernel):
    """
    The linear kernel. Functions drawn from a GP with this kernel are linear, i.e. f(x) = cx.
    The kernel equation is

        k(x, y) = σ²xy

    where σ² is the variance parameter.
    """

    def _init_(self,**kwrags):
        for kwarg in kwargs:
            if kwarg not in {"name","active_dims"}:
                raise TypeError(f"Unkown keyword a argument:{kwarg}")
        super().__init__(**kwargs)
        self.variance = Parameter(variance, transform=positive())
        self._validate_ard_active_dims(self.variance)        



    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.lengthscales.shape.ndims > 0


    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


    def R1Product(self, X):
        Xs = tf.matmul(X,self.variance)
        r1 = tf.tensordot(Xs,X,[[-1],[-1]])
        return tf.sqrt(1+r1)
       
    def R1Product(self, X2=None):
        Xs = tf.matmul(X2,self.variance)
        r3 = tf.tensordot(Xs,X2,[[-1],[-1]])
        return tf.sqrt(1+r3)
    
    def R2Product(self,X,X2=None):
        Xs = tf.matmul(X,self.variance)
        r2 = tf.tensordot(Xs,X2,[[-1],[-1]])
        return r2




class CONV(DOT):

    def K_r(self,r1,r2,r3):
        return tf.asin(2*r2/(r1*r3))