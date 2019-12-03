## Releasing a New Version

1. Bump the version in `setup.py`
1. Commit bumped version and tag with version
1. Run:
   ```shell
   bash release.sh
   ```
1. Pip install the package to verify the version was bumped:
   ```shell
   pip install skplumber
   pip freeze | grep skplumber
   ```
1. Run:
   ```shell
   git push origin <version_tag_name>
   ```
