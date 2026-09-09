# Data

This project uses the Sleep-EDF Expanded dataset from PhysioNet.

Raw EDF files are not included in this repository.

Expected local structure:

data/
└── sleep-edf/
    ├── SC4001/
    ├── SC4002/
    ├── SC4011/
    ├── SC4012/
    ├── SC4021/
    ├── SC4031/
    ├── SC4041/
    └── SC4051/

Each recording folder should contain:
- one `*-PSG.edf`
- one `*-Hypnogram.edf`