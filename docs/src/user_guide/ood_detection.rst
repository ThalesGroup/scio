Out-of-Distribution Detection
=============================

The concept
-----------

The task of « **Out-of-Distribution** Detection » (or « **OoD** Detection ») is a particular case of `novelty detection <https://en.wikipedia.org/wiki/Novelty_detection>`_. It consists in identifying, at runtime, inputs for which a model's prediction should be rejected, by lack of informed support. For example, if a Neural Network was trained to classify images of cats and dogs, it would seem irrelevant to trust its output on a tree image. Although there is no absolute definitive answer to this question (depending on the context, what a person considers OoD might be considered **In-Distribution** (**InD**) by another), it is still possible to perform deviation estimations through confidence scores.

In ``scio``, we aim at performing OoD Detection naturally by **thresholding** confidence scores. That is, defining the following decision function for a given threshold :math:`\tau`:

.. math::

	x\text{ is OoD}\Longleftrightarrow \text{Score}(x)\leqslant\tau.

Ultimately, the choice of threshold is a trade-off between False Positives and True Positives, left to the user's responsibility. However, it may be relevant to compare score functions in their ability to offer good trade-offs. For example, imagine we have at our disposal :math:`X_{\text{InD}}` and :math:`X_{\text{OoD}}` corresponding to sets of samples which we know to be respectively InD and OoD. Then, we can evaluate score functions, say :math:`S_{\text{bad}}` and :math:`S_{\text{good}}`, on those sets and plot the following histograms.

.. tikz::
   :alt: Illustrative histogram plots

   \begin{tikzpicture}[domain=-2.3:5.8, samples=100]

     % First plot (left)
     \begin{scope}
       \draw[very thin,color=gray] (-2.3, 0) grid (5.8, 3.2);
       \draw[thick, -stealth] (-2.4, 0) -- (6, 0) node[below left] {Score value for $S_{\text{bad}}$};
       \draw[thick, -stealth] (0, 0) -- (0, 3.3) node[left] {Density};

       \fill[color=myornge, fill opacity=.35] (-2.3, 0) -- plot (\x, {2.2*1.2^(-(\x -.7)*(\x -.7)}) -- (5.8, 0);
       \draw[color=myorange, thick] plot (\x, {2.2*1.2^(-(\x -.7)*(\x -.7)});
       \draw[color=myorange] (-1.7, 1.45) node {\large $X_{\text{\normalfont OoD}}$};

       \fill[color=mygreen, fill opacity=.35] (-2.3, 0) -- plot (\x, {2.5*1.3^(-(\x -1.2)*(\x -1.2)}) -- (5.8, 0);
       \draw[color=mygreen, opacity=.8, thick] plot (\x, {2.5*1.3^(-(\x -1.2)*(\x -1.2)});
       \draw[color=mygreen] (2.55, 2.3) node {\large $X_{\text{\normalfont InD}}$};
     \end{scope}

     % Second plot (right), shifted
     \begin{scope}[xshift=10cm]
       \draw[very thin,color=gray] (-2.3, 0) grid (5.8, 3.2);
       \draw[thick, -stealth] (-2.4, 0) -- (6, 0) node[below left] {Score value for $S_{\text{good}}$};
       \draw[thick, -stealth] (0, 0) -- (0, 3.3) node[right] {Density};

       \fill[color=myorange, fill opacity=.35] (-2.3, 0) -- plot (\x, {2.6*1.8^(-(\x +.6)*(\x +.6)}) -- (5.8, 0);
       \draw[color=myorange, thick] plot (\x, {2.6*1.8^(-(\x +.6)*(\x +.6)});
       \draw[color=myorange] (-1.65, 2.5) node {\large $X_{\text{\normalfont OoD}}$};

       \fill[color=mygreen, fill opacity=.35] (-2.3, 0) -- plot (\x, {2.4*1.5^(-(\x -3.2)*(\x -3.2)}) -- (5.8, 0);
       \draw[color=mygreen, thick] plot (\x, {2.4*1.5^(-(\x -3.2)*(\x -3.2)});
       \draw[color=mygreen] (4.35, 2.25) node {\large $X_{\text{\normalfont InD}}$};
     \end{scope}

   \end{tikzpicture}

In this case, it seems clear that for the chosen InD and OoD representatives, :math:`S_{\text{good}}` offers better trade-offs than :math:`S_{\text{bad}}`. In ``scio.eval``, we provide tools and metrics to :doc:`visualize and quantify <../api_references/eval>` such observations.

Application
-----------

The ability to robustly identify Out-of-Distribution samples may have many useful applications. Simple examples are:

#. **Monitoring.** Using the OoD Detection as a **reject option** may help using safe callbacks more efficiently for deployed models.
#. **Training.** Some training procedures may benefit from identifying samples for which a model is currently unfit to infer.
