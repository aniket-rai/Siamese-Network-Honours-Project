# Things I've gathered about this whole business

I'm working through [this
tutorial](https://www.javatpoint.com/opencv-basic-operation-on-images)

I'm using the `opencv-test` conda environment.

So here's the thing... OpenCV is a massive package - pretty much anything you
want to do with Computer Vision can be done with OpenCV. So I kind of stopped
following along once I realised how much stuff the tutorial was going to cover.
I think, if we're wanting to use some functionality from it, we'd be better
served searching particular things instead of trying to learn the whole library.

- Getting anaconda to work on WSL, while a little fiddly, is possible, if you
  follow [these steps](https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da)
- The image displaying stuff doesn't work on WSL, which is unfortunate. The good
  thing is that you can create the environment in Windows, and it will work
  there.
- The key is to export the environment *specifically* with the `--from-history`
  flag. Therefore, the steps are:
  1. Write your code in WSL, if that's where you're going to be.
     - It might make sense to just turn Vim mode back on in VSCode, and stay on
       the Windows side for the time being. That, or just only use linux for
       vim.
  2. When you want to work on it from Windows, run `conda env export --from
     history > environment.yml`
  3. On the Windows side, run `conda env create` to install the environment.
- The [cheatsheet](file:///C:/Users/64221/Downloads/conda-cheatsheet.pdf) is
  vitally important.
