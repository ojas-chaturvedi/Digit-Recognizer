% Required for TeX classes
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{notes}[Custom notes class]

% Based off of the `article' class
\LoadClass[12pt]{article}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Package options and document metadata
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[letterpaper]{geometry}
% \usepackage{geometry}
\geometry{letterpaper, portrait, margin=1in}
\RequirePackage{microtype}
\RequirePackage{enumerate}
\RequirePackage{multicol}

% Skip lines and don't indent
\setlength{\parindent}{0em}
\setlength{\parskip}{1em}

% Import math fonts and symbols
\RequirePackage{amsfonts}
\RequirePackage{amsmath,amsthm,amssymb}

% Palatino typeface -- includes (old-style) text figures
% \RequirePackage[osf]{mathpazo}

\RequirePackage{fullpage}
\RequirePackage{epic}
\RequirePackage{eepic}
\RequirePackage{graphicx}
\RequirePackage{shortcuts}

\setlength{\oddsidemargin}{0in}
\setlength{\topmargin}{0in}
\setlength{\textwidth}{6.5in}
\setlength{\textheight}{8.5in}

\setlength{\fboxrule}{.5mm}\setlength{\fboxsep}{1.2mm}
\newlength{\boxlength}\setlength{\boxlength}{\textwidth}
\addtolength{\boxlength}{-4mm}

\newcommand{\handout}[5]{
    \noindent
    \begin{center}
    \framebox{
    \vbox{
        \hbox to 5.78in { {\bf Capstone: The Art of Approximation } \hfill #2 }
        \vspace{4mm}
        \hbox to 5.78in { {\Large \hfill #5  \hfill} }
        \vspace{2mm}
        \hbox to 5.78in { {\em #3 \hfill #4} }
    }
    }
    \end{center}
    \vspace*{4mm}
}

\newcommand{\lecture}[4]{\handout{#1}{#2}{#3}{#4}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Custom commands -- words, environments, and formatting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Boxed ``to-do'' statements to make incomplete problems clear
\newcommand{\todo}{\fbox{TO-DO}\ \ }

% 75% measure rules to split problems into sections
\newcommand{\separator}{\vfill \begin{center} \rule{0.75\linewidth}{0.5pt} \\ \end{center} \vfill}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Custom commands -- redefined commands that should look or behave differently
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Redefine epsilon and empty set commands to use better-looking versions
\renewcommand{\epsilon}{\ensuremath{\varepsilon}}
\renewcommand{\emptyset}{\varnothing}

% Redefine the bar and vec commands to use larger overlines and bold-faced vectors
\renewcommand{\bar}{\overline}
\renewcommand{\vec}{\mathbf}
