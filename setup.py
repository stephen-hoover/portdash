from setuptools import find_packages, setup


def main():
    setup(
        name="portdash",
        version="0.1.0",
        author="Stephen Hoover",
        author_email="Stephen.LD.Hoover@gmail.com",
        packages=find_packages(),
        install_requires=[
            'pyyaml',
            'pandas',
            'numpy',
            'dash',
            'requests',
            'flask',
            'flask-sqlalchemy',
            'flask-migrate',
        ],
    )


if __name__ == "__main__":
    main()
