import numpy as np


class Features_C123:
    def __init__(self, info: list):
        class MyFeature():
            def __init__(self, pkl_files, allow_empty=False):
                self.data_loaded = False
                self.data_files = pkl_files
                self.allow_empty = allow_empty  # host only feature,
                self.data = None
                self.data_shape = None

            def __getitem__(self, index):
                if not self.data_loaded:
                    self.data = dict()
                    for pkl_file in self.data_files:
                        self.data.update(np.load(pkl_file, allow_pickle=True))
                    self.data_loaded = True
                    self.data_shape = list(self.data.values())[0].shape
                    if 'average' not in self.data.keys():
                        self.data['average'] = np.array(list(self.data.values())).mean(0)

                if not self.allow_empty:
                    v = self.data[index]
                else:
                    if index in self.data.keys():
                        v = self.data[index]
                    else:
                        v = self.data['average']
                return v

        feature_EsmMean = MyFeature(["../features/v_and_h_esm2.pkl"])  # shape(2560,)
        feature_prottrans = MyFeature(["../features/v_and_h_prottrans.pkl"])  # shape (1024,)
        feature_doc2vec = MyFeature(["../features/v_and_h_doc2vec.pkl"])  # shape (32, )

        feature_aac = MyFeature(["../features/v_and_h_AAC.pkl"])  # shape (20,)
        feature_paac = MyFeature(["../features/v_and_h_PAAC.pkl"])  # shape (50, )
        feature_apaac = MyFeature(["../features/v_and_h_APAAC.pkl"])  # shape (80,)
        feature_dc = MyFeature(["../features/v_and_h_DC.pkl"])  # shape (400, )
        feature_ct = MyFeature(["../features/v_and_h_CT.pkl"])  # shape (343, )
        feature_cksaap = MyFeature(["../features/v_and_h_CKSAAP.pkl"])  # shape (1200,)
        feature_tc = MyFeature(["../features/v_and_h_TC.pkl"])  # shape (8000, )
        feature_ctdc = MyFeature(["../features/v_and_h_CTDC.pkl"])  # shape (21, )
        feature_ctdd = MyFeature(["../features/v_and_h_CTDD.pkl"])  # shape (105, )
        feature_ctdt = MyFeature(["../features/v_and_h_CTDT.pkl"])  # shape (21, )
        feature_geary = MyFeature(["../features/v_and_h_Geary.pkl"])  # shape (240, )
        feature_moran = MyFeature(["../features/v_and_h_Moran.pkl"])  # shape (240, )
        feature_moreaubroto = MyFeature(["../features/v_and_h_MoreauBroto.pkl"])  # shape (240, )
        feature_qso = MyFeature(["../features/v_and_h_QSO.pkl"])  # shape (100, )
        feature_socn = MyFeature(["../features/v_and_h_SOCN.pkl"])  # shape (60, )

        feature_aac_pssm = MyFeature(["../features/v_and_h_AAC-PSSM.pkl"])  # shape (20, )
        feature_dpc_pssm = MyFeature(["../features/v_and_h_DPC-PSSM.pkl"])  # shape (400, )
        feature_rpssm = MyFeature(["../features/v_and_h_RPSSM.pkl"])  # shape (110, )
        feature_pssmac = MyFeature(["../features/v_and_h_PSSMAC.pkl"])  # shape (1200, )

        feature_HNetStruc2vec = MyFeature(["../features/Human_NetNode2vec.pkl"], allow_empty=True)  # shape(256,)
        feature_HNetNode2vec = MyFeature(["../features/Human_NetStruc2vec.pkl"], allow_empty=True)  # shape(256,)
        feature_HNetTP = MyFeature(["../features/Human_NetTP.pkl"], allow_empty=True)  # shape(8,)

        self.info = info
        self.features = {
            'aac': feature_aac,
            'dc': feature_dc,
            'tc': feature_tc,
            'cksaap': feature_cksaap,

            'apaac': feature_apaac,
            'paac': feature_paac,
            'ct': feature_ct,
            'ctdc': feature_ctdc,
            'ctdt': feature_ctdt,
            'ctdd': feature_ctdd,
            'geary': feature_geary,
            'moran': feature_moran,
            'moreaubroto': feature_moreaubroto,
            'qso': feature_qso,
            'socn': feature_socn,

            "aac_pssm": feature_aac_pssm,
            "dpc_pssm": feature_dpc_pssm,
            "rpssm": feature_rpssm,
            "pssmac": feature_pssmac,

            'EsmMean': feature_EsmMean,
            'prottrans': feature_prottrans,
            'doc2vec': feature_doc2vec,

            'HNetNode2vec': feature_HNetStruc2vec,
            'HNetStruc2vec': feature_HNetNode2vec,
            'HNetTP': feature_HNetTP,

        }

    def get(self, index, foldn=None):
        return np.hstack([self.features[i][index] for i in self.info])

    def __getitem__(self, index):
        return np.hstack([self.features[i][index] for i in self.info])
