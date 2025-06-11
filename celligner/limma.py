###########################################################
#
# limmapy
#
##################################################################

from __future__ import print_function
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

# Import R package
limma = importr('limma')

# Optional utility to force R conversion to data.frame
to_dataframe = robjects.r('function(x) data.frame(x)')

class limmapy:
    '''
    limma wrapper using rpy2

    Args:
        count_matrix: pandas DataFrame with gene IDs as index, samples as columns
        design_matrix: pandas DataFrame with samples as rows, design variables as columns
    '''

    def __init__(self):
        self.limma_result = None

    def lmFit(self, count_matrix, design_matrix, **kwargs):
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_counts = ro.conversion.py2rpy(count_matrix.astype(int))
            r_design = ro.conversion.py2rpy(design_matrix.astype(int))
        self.fit = limma.lmFit(r_counts, r_design, **kwargs)
        return self

    def eBayes(self, **kwargs):
        self.fit = limma.eBayes(self.fit, **kwargs)
        return self

    def topTable(self, **kwargs):
        val = limma.topTable(self.fit, **kwargs)
        if isinstance(val, robjects.vectors.DataFrame):
            with localconverter(ro.default_converter + pandas2ri.converter):
                val = ro.conversion.rpy2py(val)
        return val