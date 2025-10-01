import os
from setuptools import setup, find_packages

# 读取README内容
with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

# 读取依赖列表
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='sar-object-detection',
    version='0.1.0',
    description='SAR图像多类别有向目标检测项目',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='SAR Detection Team',
    author_email='sar_detection@example.com',
    url='https://github.com/sar-detection-team/sar-object-detection',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'sar-train=tools.train:main',
            'sar-test=tools.test:main',
            'sar-infer=tools.inference:main',
            'sar-prepare-data=scripts.prepare_data:main',
            'sar-demo=scripts.demo:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)