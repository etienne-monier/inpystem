=================================================================================
                             ITKrMM - Penknife
=================================================================================

This is a small collection of Matlab m-files that implement the ITKrMM 
(iterative thresholding& K residual means for masked data) algorithm 
for dictionary learning from incomplete data and the low rank atom recovery 
algorithm from incomplete data as described in:

’Dictionary learning from incomplete data’ V.Naumova, K Schnass, arXiv:1701.03655
In particular it can be used to reproduce the figures/experiments in the paper.

We do not guarantee correctness especially not for wksvd 
but are happy about any feedback!  

Karin & Valeriya
(karin.schnass@uibk.ac.at, valeriya@simula.no)
=================================================================================

What's in it:
=================================================================================

1. itkrmm.m - implements the ITKrMM algorithm. 
	Input parameters are described inside.

2. rec_lratom - implements the algorithm for recovering a low rank atom from masked data.	
	Input parameters are described inside.

3. itkrm.m - implements the ITKrM algorithm on which ITKrMM is based (for comparison). 
	Input parameters are described inside.

4. makesparselowsig.m - to generate training signals with a sparse and a low rank 	
	component with various parameters described inside.

5. RandMask.m - generates the random erasure masks with varying erasure probabilities
	as described in the paper, input parameters described inside.

6. maskTimeVar.m - generates the burst error mask with varying locations and burst 
	lengths as described in the paper, input parameters described inside.

7. test_script.m - to reproduce the synthetic experiments in the paper, 
	can be run as is using default settings.

8. inpainting_script.m - to reproduce the inpainting experiments in the paper, 
	can be run as is using default settings.
 
9. pic2patches.m - extracts patches and locations from an image, 
	input parameters described inside.

10. patches2pic.m - recomposes patches and locations to an image,
	input parameters described inside.

11. showdico.m - converts a vectorised patch dictionary to a 2-dimensional 
	image of the patch dictionary, usage described inside. 

12. OMPm.m - implements Orthogonal Matching Pursuit for masked signals, as used
	for inpainting and inside wKSVD, input parameters described inside.

13. wKSVD.m - caveat emptor!!! - implementation of the weighted KSVD algorithm
	in its version adapted to masked data with fixed sparsity level, 
	as described in:
	’Sparse Representation for Color Image Restoration’
	by J. Mairal, M. Elad and G. Sapiro, IEEE TSP, 2008,
	input parameters described inside.

14. images - a folder containing all the test images and the cracks mask.

=================================================================================

