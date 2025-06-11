from celligner.params import *
from celligner import limma

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import umap.umap_ as umap

import scanpy as sc
from anndata import AnnData

import os
import pickle
import gc

import pandas as pd
import numpy as np

#from contrastive import CPCA
from mnnpy import mnn

class Celligner(object):
    def __init__(
        self,
        topKGenes=TOP_K_GENES,
        pca_ncomp=PCA_NCOMP,
        cpca_ncomp=CPCA_NCOMP,
        louvain_kwargs=LOUVAIN_PARAMS,
        mnn_kwargs=MNN_PARAMS,
        umap_kwargs=UMAP_PARAMS,
        mnn_method="mnn_marioni",
        low_mem=False,
    ):
        """
        Initialize Celligner object

        Args:
            topKGenes (int, optional): see params.py. Defaults to 1000.
            pca_ncomp (int, optional): see params.py. Defaults to 70.
            cpca_ncomp (int, optional): see params.py. Defaults to 4.
            louvain_kwargs (dict, optional): see params.py
            mnn_kwargs (dict, optional): see params.py 
            umap_kwargs (dict, optional): see params.py
            mnn_method (str, optional): Only default "mnn_marioni" supported right now.
            low_mem (bool, optional): adviced if you have less than 32Gb of RAM. Defaults to False.
        """
        
        self.topKGenes = topKGenes
        self.pca_ncomp = pca_ncomp
        self.cpca_ncomp = cpca_ncomp
        self.louvain_kwargs = louvain_kwargs
        self.mnn_kwargs = mnn_kwargs
        self.umap_kwargs = umap_kwargs
        self.mnn_method = mnn_method
        self.low_mem = low_mem

        self.ref_input = None
        self.ref_clusters = None
        self.ref_de_genes = None
        
        self.target_input = None
        self.centered_target_input = None
        self.target_clusters = None
        self.target_de_genes = None

        self.de_genes = None
        self.cpca_loadings = None
        self.cpca_explained_var = None
        self.combined_output = None
        
        self.umap_reduced = None
        self.output_clusters = None
        self.tumor_CL_dist = None


    def __checkExpression(self, expression, is_reference):
        """
        Checks gene overlap with reference, checks for NaNs, then does mean-centering.

        Args:
            expression (pd.Dataframe): expression data as samples (rows) x genes (columns)
            is_reference (bool): whether the expression is a reference or target

        Raises:
            ValueError: if some common genes are missing from the expression dataset
            ValueError: if the expression matrix contains nan values

        Returns:
            (pd.Dataframe): the expression matrix
        """
        # Check gene overlap
        if expression.loc[:, expression.columns.isin(self.common_genes)].shape[1] < len(self.common_genes):
            if not is_reference:
                raise ValueError("Some genes from reference dataset not found in target dataset")
            else:
                raise ValueError("Some genes from previously fit target dataset not found in new reference dataset")
        
        expression = expression.loc[:, self.common_genes].astype(float)
        
        # Raise issue if there are any NaNs in the expression dataframe
        if expression.isnull().values.any():
            raise ValueError("Expression dataframe contains NaNs")

        # Mean center the expression dataframe
        # expression = expression.sub(expression.mean(0), 1)
        
        return expression
    
    def __meanCenter(self):
        """
        Centers the target expression data by subtracting the mean of each gene across all samples and adding the mean of the reference expression data.
        This is done to ensure that the target expression data is centered around the reference expression data.

        Returns:
            pd.DataFrame: centered target expression dataframe
        """
        # Center the target expression data
        centered_target_input = self.target_input - self.target_input.mean(0)
        
        # Add the mean of the reference expression data
        centered_target_input += self.ref_input.mean(0)
        
        return centered_target_input


    def __cluster(self, expression):
        """
        Cluster expression in (n=70)-dimensional PCA space using a shared nearest neighbor based method

        Args:
            expression (pd.Dataframe): expression data as samples (rows) x genes (columns)

        Returns:
            (list): cluster label for each sample
        """
        # Create anndata object
        adata = AnnData(expression, dtype='float64')

        # Find PCs
        print("Doing PCA..")
        sc.tl.pca(adata, n_comps=self.pca_ncomp, zero_center=True, svd_solver='arpack')

        # Find shared nearest neighbors (SNN) in PC space
        # Might produce different results from the R version as ScanPy and Seurat differ in their implementation.
        print("Computing neighbors..")
        sc.pp.neighbors(adata, knn=True, use_rep='X_pca', n_neighbors=20, n_pcs=self.pca_ncomp)
        
        print("Clustering..")
        sc.tl.louvain(adata, use_weights=True, **self.louvain_kwargs)
        fit_clusters = adata.obs["louvain"].values.astype(int)
        
        del adata
        gc.collect()

        return fit_clusters


    def __runDiffExprOnClusters(self, expression, clusters):
        """
        Runs limma (R) on the clustered data.

        Args:
            expression (pd.Dataframe): expression data
            clusters (list): the cluster labels (per sample)

        Returns:
            (pd.Dataframe): limmapy results
        """

        n_clusts = len(set(clusters))
        print("Running differential expression on " + str(n_clusts) + " clusters..")
        clusts = set(clusters) - set([-1])
        
        # make a design matrix
        design_matrix = pd.DataFrame(
            index=expression.index,
            data=np.array([clusters == i for i in clusts]).T,
            columns=["C" + str(i) + "C" for i in clusts],
        )
        design_matrix.index = design_matrix.index.astype(str).str.replace("-", ".")
        design_matrix = design_matrix[design_matrix.sum(1) > 0]
        
        # creating the matrix
        data = expression.T
        data = data[data.columns[clusters != -1].tolist()]
        
        # running limmapy
        print("Running limmapy..")
        res = (
            limma.limmapy()
            .lmFit(data, design_matrix)
            .eBayes(trend=False)
            .topTable(number=len(data)) 
            .iloc[:, len(clusts) :]
        )
        return res.sort_values(by="F", ascending=False)
    

    def __runCPCA(self, centered_ref_input, centered_target_input):
        """
        Perform contrastive PCA on the centered reference and target expression datasets

        Args:
            centered_ref_input (pd.DataFrame): reference expression matrix where the cluster mean has been subtracted
            centered_target_input (pd.DataFrame): target expression matrix where the cluster mean has been subtracted

        Returns:
            (ndarray, ncomponents x ngenes): principal axes in feature space
            (ndarray, ncomponents,): variance explained by each component

        """
        target_cov = centered_target_input.cov()
        ref_cov = centered_ref_input.cov()
        if not self.low_mem:
            pca = PCA(self.cpca_ncomp, svd_solver="randomized", copy=False)
        else: 
            pca = IncrementalPCA(self.cpca_ncomp, copy=False, batch_size=1000)
        
        pca.fit(target_cov - ref_cov)
        return pca.components_, pca.explained_variance_
    
    def __apply_cpca_shift(self, X_ref, X_target, cpca_loadings):
        """
        Align target samples to reference by shifting their cPC projections to match the reference centroid.

        Args:
            X_ref (pd.DataFrame): Reference expression (samples × genes)
            X_target (pd.DataFrame): Target expression (samples × genes)
            cpca_loadings (np.ndarray): cPCA loadings (n_cpcs × n_genes)

        Returns:
            pd.DataFrame: Shift-corrected target expression (same shape as X_target)
        """
        assert list(X_ref.columns) == list(X_target.columns), "Mismatched gene order"
        assert cpca_loadings.shape[1] == X_target.shape[1], "cPCA loadings and input gene dims must match"

        # Projection into cPC space
        Z_ref = cpca_loadings @ X_ref.T.values
        Z_target = cpca_loadings @ X_target.T.values

        # Compute shift
        delta = Z_ref.mean(axis=1).reshape(-1, 1) - Z_target.mean(axis=1).reshape(-1, 1)

        # Apply shift
        Z_target_shifted = Z_target + delta

        # Back-project
        X_target_corrected = (cpca_loadings.T @ Z_target_shifted).T

        # Rewrap into DataFrame
        return pd.DataFrame(X_target_corrected, index=X_target.index, columns=X_target.columns)

    def fit(self, ref_expr):
        """
        Fit the model to the reference expression dataset - cluster + find differentially expressed genes.

        Args:
            ref_expr (pd.Dataframe): reference expression matrix of samples (rows) by genes (columns), 
                where genes are ensembl gene IDs. Data should be log2(X+1) TPM data. 
                In the standard Celligner pipeline this the cell line data.

        Raises:
                ValueError: if only 1 cluster is found in the PCs of the expression
        """
        
        self.common_genes = list(ref_expr.columns)
        self.ref_input = self.__checkExpression(ref_expr, is_reference=True)
        
        # Cluster and find differential expression for reference data
        self.ref_clusters = self.__cluster(self.ref_input)
        if len(set(self.ref_clusters)) < 2:
            raise ValueError("Only one cluster found in reference data, no differential expression possible")
        self.ref_de_genes = self.__runDiffExprOnClusters(self.ref_input, self.ref_clusters)

        return self


    def transform(self, target_expr=None, compute_cPCs=True):
        """
        Align samples in the target dataset to samples in the reference dataset

        Args:
            target_expr (pd.Dataframe, optional): target expression matrix of samples (rows) by genes (columns), 
                where genes are ensembl gene IDs. Data should be log2(X+1) TPM data.
            compute_cPCs (bool, optional): if True, compute cPCs from the fitted reference and target expression. Defaults to True.

        Raises:
            ValueError: if compute_cPCs is True but there is no reference input (fit has not been run)
            ValueError: if compute_cPCs is False but there are no previously computed cPCs available (transform has not been previously run)
            ValueError: if no target expression is provided and there is no previously provided target data
            ValueError: if no target expression is provided and compute_cPCs is true; there is no use case for this
            ValueError: if there are not enough clusters to compute DE genes for the target dataset
        """

        if self.ref_input is None and compute_cPCs:
            raise ValueError("Need fitted reference dataset to compute cPCs, run fit function first")

        if not compute_cPCs and self.cpca_loadings is None:
            raise ValueError("No cPCs found, transform needs to be run with compute_cPCs==True at least once")

        if target_expr is None and self.target_input is None:
            raise ValueError("No previous data found for target, transform needs to be run with target expression at least once")

        if not compute_cPCs and target_expr is None:
            raise ValueError("No use case for running transform without new target data when compute_cPCs==True")

        if compute_cPCs:

            if target_expr is not None:

                self.target_input = self.__checkExpression(target_expr, is_reference=False)
                self.centered_target_input = self.__meanCenter()

                # Cluster and find differential expression for target data
                self.target_clusters = self.__cluster(self.centered_target_input)
                if len(set(self.target_clusters)) < 2:
                    raise ValueError("Only one cluster found in target data, no DE possible")
                self.target_de_genes = self.__runDiffExprOnClusters(self.centered_target_input, self.target_clusters)

                self.de_genes = pd.Series(list(self.ref_de_genes[:self.topKGenes].index) +
                                          list(self.target_de_genes[:self.topKGenes].index)).drop_duplicates().tolist()

            else:
                print("INFO: No new target expression provided, using previously provided target dataset")

            # Subtract cluster average from cluster samples
            cluster_centered_ref = pd.concat(
                [
                    self.ref_input.loc[self.ref_clusters == val] - self.ref_input.loc[self.ref_clusters == val].mean(axis=0)
                    for val in set(self.ref_clusters)
                ]
            ).loc[self.ref_input.index]
            
            cluster_centered_target = pd.concat(
                [
                    self.target_input.loc[self.target_clusters == val] - self.target_input.loc[self.target_clusters == val].mean(axis=0)
                    for val in set(self.target_clusters)
                ]
            ).loc[self.target_input.index]

            print("Running cPCA...")
            self.cpca_loadings, self.cpca_explained_var = self.__runCPCA(cluster_centered_ref, cluster_centered_target)

            del cluster_centered_ref, cluster_centered_target
            gc.collect()

        else:
            missing_genes = list(self.ref_input.loc[:, ~self.ref_input.columns.isin(target_expr.columns)].columns)
            if len(missing_genes) > 0:
                print('WARNING: %d genes from reference dataset not found in new target dataset, subsetting to overlap' % (len(missing_genes)))
                drop_idx = [self.ref_input.columns.get_loc(g) for g in missing_genes]
                self.ref_input = self.ref_input.loc[:, self.ref_input.columns.isin(target_expr.columns)]
                self.common_genes = list(self.ref_input.columns)
                self.cpca_loadings = np.array([np.delete(self.cpca_loadings[n], drop_idx) for n in range(self.cpca_ncomp)])
                overlap = self.ref_input.loc[:, self.ref_input.columns.isin(self.de_genes)]
                if overlap.shape[1] < len(self.de_genes):
                    print('WARNING: dropped genes include %d differentially expressed genes that may be important' % (len(self.de_genes) - overlap.shape[1]))
                    temp = pd.Series(self.de_genes)
                    self.de_genes = temp[temp.isin(self.ref_input.columns)].to_list()

            self.target_input = self.__checkExpression(target_expr, is_reference=False)
            self.centered_target_input = self.__meanCenter()

        # print("Computing cPC shift for target alignment...")
        # X_ref = self.ref_input
        # X_target = self.centered_target_input

        # print(f"dimension of X_target: {X_target.shape}, dimension of X_ref: {X_ref.shape}")

        # print(f"shape of cpca_loadings: {self.cpca_loadings.shape}, n_components: {self.cpca_ncomp}, n_genes: {self.cpca_loadings.shape[1]}")

        # Step 2: Fit regressions
        # reg_target = LinearRegression(fit_intercept=False).fit(self.cpca_loadings.T, X_target.T)
        # reg_ref = LinearRegression(fit_intercept=False).fit(self.cpca_loadings.T, X_ref.T)

        # # Step 3: Predict component signals
        # fitted_target = reg_target.predict(self.cpca_loadings.T).T
        # fitted_ref = reg_ref.predict(self.cpca_loadings.T).mean(axis=1, keepdims=True)

        # # check shapes
        # print(f"fitted_target shape: {fitted_target.shape}, fitted_ref shape: {fitted_ref.shape}")

        # fitted_ref_df = pd.DataFrame(
        #     np.repeat(fitted_ref.T, repeats=X_target.shape[0], axis=0),
        #     index=X_target.index,
        #     columns=X_target.columns
        # )
        # print(f"fitted_ref_df shape: {fitted_ref_df.shape}")

        # # Step 4: Subtract target's signal and add back ref's average signal
        # X_target_corrected = X_target - fitted_target + fitted_ref_df
        # display(X_target_corrected)

        X_target_corrected = self.__apply_cpca_shift(self.ref_input, self.centered_target_input, self.cpca_loadings)
        print("Corrected target expression matrix:")
        display(X_target_corrected)
        print('input ref expression matrix:')
        display(self.ref_input)

        print("Doing the MNN analysis using Marioni et al. method...")
        varsubset = np.array([1 if i in self.de_genes else 0 for i in self.centered_target_input.columns]).astype(bool)
        print(self.ref_input.shape, X_target_corrected.shape, varsubset.sum(), self.mnn_kwargs)
        print("ref_input shape:", self.ref_input.shape)
        print("X_target_corrected shape:", X_target_corrected.shape)
        print("varsubset shape:", varsubset.shape)
        print("var_index:", len(list(range(len(self.ref_input.columns)))))
        assert (self.ref_input.columns == X_target_corrected.columns).all(), "Reference and target expression matrices do not have the same genes"
        target_corrected, self.mnn_pairs = mnn.marioniCorrect(
            self.ref_input,
            X_target_corrected,
            var_index=list(range(len(self.ref_input.columns))),
            var_subset=varsubset,
            **self.mnn_kwargs,
        )

        self.combined_output = pd.concat([self.ref_input, target_corrected])
        del target_corrected
        gc.collect()

        print('Done')
        return self

    def computeMetricsForOutput(self, umap_rand_seed=14, UMAP_only=False, model_ids=None, tumor_ids=None):
        """
        Compute UMAP embedding and optionally clusters and tumor - model distance.
        
        Args:
            UMAP_only (bool, optional): Only recompute the UMAP. Defaults to False.
            umap_rand_seed (int, optional): Set seed for UMAP, to try an alternative. Defaults to 14.
            model_ids (list, optional): model IDs for computing tumor-CL distance. Defaults to None, in which case the reference index is used.
            tumor_ids (list, optional): tumor IDs for computing tumor-CL distance. Defaults to None, in which case the target index is used.
        
        Raises:
            ValueError: if there is no corrected expression matrix
        """
        if self.combined_output is None:
            raise ValueError("No corrected expression matrix found, run this function after transform()")

        print("Computing UMAP embedding...")
        # Compute UMAP embedding for results
        pca = PCA(self.pca_ncomp)
        pcs = pca.fit_transform(self.combined_output)
        
        umap_reduced = umap.UMAP(**self.umap_kwargs, random_state=umap_rand_seed).fit_transform(pcs)
        self.umap_reduced = pd.DataFrame(umap_reduced, index=self.combined_output.index, columns=['umap1','umap2'])

        if not UMAP_only:
            
            print('Computing clusters..')
            self.output_clusters = self.__cluster(self.combined_output)

            print("Computing tumor-CL distance..")
            pcs = pd.DataFrame(pcs, index=self.combined_output.index)
            if model_ids is None: model_ids = self.ref_input.index
            if tumor_ids is None: tumor_ids = self.target_input.index
            model_pcs = pcs[pcs.index.isin(model_ids)]
            tumor_pcs = pcs[pcs.index.isin(tumor_ids)]
            
            self.tumor_CL_dist = pd.DataFrame(metrics.pairwise_distances(tumor_pcs, model_pcs), index=tumor_pcs.index, columns=model_pcs.index)
        
        return self


    def makeNewReference(self):
        """
        Make a new reference dataset from the previously transformed reference+target datasets. 
        Used for multi-dataset alignment with previously computed cPCs and DE genes.
        
        """
        self.ref_input = self.combined_output
        self.target_input = None
        return self
    
    
    def save(self, file_name):
        """
        Save the model as a pickle file

        Args:
            file_name (str): name of file in which to save the model
        """
        # save the model
        with open(os.path.normpath(file_name), "wb") as f:
            pickle.dump(self, f)


    def load(self, file_name):
        """
        Load the model from a pickle file

        Args:
            file_name (str): pickle file to load the model from
        """
        with open(os.path.normpath(file_name), "rb") as f:
            model = pickle.load(f)
            self.__dict__.update(model.__dict__)
        return self