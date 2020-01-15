Introduction
============

What is inpystem
--------------

inpystem is an open source Python library which provides tools to reconstruct partially sampled 2D images as multi-band images.

inpystem's core is a set of reconstruction techniques such as interpolation, regularized least-square and dictionary learning methods. It provides a user interface which simplify the use of these techniques.

inpystem is mainly at the destination of the microscopy community so that it highly depends on the good library HyperSpy_.

This library was originally developed by its creator Etienne Monier to handle EELS data and develop reconstruction algorithms. This was proposed afterwards to the microscopy community as a tool. 

.. _HyperSpy: https://hyperspy.org/

About the developer
-------------------

This library is developed by Etienne Monier, a French PhD student.

His research interests are in new methods and algorithms to solve challenging acquisition problems encountered in the acquisition of multi-band microscopy images. In particular, he is interested in acquiring high-SNR Electron Energy Loss Spectroscopy (EELS) images with extremely low beam energy to prevent destruction of sensitive microscopy samples. His goal is to provide to the EELS community precise algorithms to detect rapidly the presence of a chemical element inside a sample to reduce irradiation as much as possible. Such problem requires **signal processing** methods, such as **convex optimization** and **proximal splitting** methods.


