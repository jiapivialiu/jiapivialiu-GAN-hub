# GAN Hub - Introduction to Generative Adversarial Networks

My GAN documentation and presentation materials stay here.

## 🎯 View the Slides Online

**Live Presentation**: [https://jiapivialiu.github.io/jiapivialiu-GAN-hub/](https://jiapivialiu.github.io/jiapivialiu-GAN-hub/)

The slides include:
- Interactive Python visualizations
- GAN training dynamics
- Applications and use cases
- Implementation examples

## 📚 Resources

Check out the original GAN paper [here](https://arxiv.org/pdf/1406.2661).

## 🛠️ Local Development

### Quick Start
```bash
# Install dependencies
make install-deps

# Build slides locally
make slides

# Preview slides in browser
make preview

# Build for GitHub Pages
make github-pages
```

### Requirements
- Python 3.11+
- Quarto
- Virtual environment support

## 📁 Project Structure

```
├── doc/slides/           # Source Quarto presentation
├── index.html           # GitHub Pages entry point
├── gan_introduction_files/ # Presentation assets
├── .github/workflows/   # Auto-deployment
└── Makefile            # Build automation
```

## 🚀 Deployment

The slides are automatically deployed to GitHub Pages when you push to the main branch. The GitHub Actions workflow:

1. Sets up Python and Quarto
2. Installs dependencies
3. Builds the presentation
4. Deploys to GitHub Pages

Manual deployment: `make github-pages`
