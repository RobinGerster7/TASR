import numpy as np


class CTTD:
    def forward(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Forward pass for the Chessboard-Based Hyperspectral Target Detection method.

        Args:
            img (np.ndarray): Input hyperspectral image of shape (1, C, H, W).
            target (np.ndarray): Target vector of shape (1, C, 1, 1).

        Returns:
            np.ndarray: Detection result of shape (1, 1, H, W).
        """
        B, C, H, W = img.shape
        assert B == 1, "Only B=1 is supported"

        pixel_num = H * W
        X_cube = img.reshape(C, pixel_num)  # Reshape to (C, pixel_num)
        target = target.reshape(C)  # Reshape target to (C,)

        ver, hor = 13, 2  # Fixed values as specified

        # Construct the chessboard-shaped topological framework
        chessboard_cenvalue1 = []
        target_index_set = []
        target_card_set = []

        for i in range(C):
            X_dim = X_cube[i, :]
            hist_counts, bin_edges = np.histogram(X_dim, bins=ver)
            X_dim_cen_i = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers
            chessboard_cenvalue1.append(X_dim_cen_i)

            target_val = target[i]
            index_target = np.argmin(np.abs(X_dim_cen_i - target_val))
            target_j_cardinality = hist_counts[index_target]

            if target_j_cardinality == 0:
                target_j_cardinality = 1

            target_index_set.append(index_target)
            target_card_set.append(target_j_cardinality)

        chessboard_cenvalue1 = np.array(chessboard_cenvalue1)
        target_index_set = np.array(target_index_set)
        target_card_set = np.array(target_card_set)

        # Select the optimal separable bands
        band_dis_target, bin_edges = np.histogram(target, bins=hor)
        chessboard_cenvalue_target = (bin_edges[:-1] + bin_edges[1:]) / 2
        optimal_separable_bands = []

        for k in range(hor):
            band_dis_target_k = band_dis_target[k]
            chessboard_cenvalue_target_k = chessboard_cenvalue_target[k]
            target_index_k = np.argsort(np.abs(target - chessboard_cenvalue_target_k))

            if band_dis_target_k != 0:
                target_index_k = target_index_k[:band_dis_target_k]
                min_po_target = np.argmin(target_card_set[target_index_k])
                min_band_target = target_index_k[min_po_target]
                optimal_separable_bands.append(min_band_target)

        optimal_separable_bands = np.array(optimal_separable_bands)
        target_OBJ_index_set = target_index_set[optimal_separable_bands]
        OBJ = len(optimal_separable_bands)

        # Perform the information retrieval task
        result = np.zeros(pixel_num)

        for i in range(pixel_num):
            x = X_cube[:, i]
            x_OBJ_index_set = []

            for n in range(OBJ):
                j = optimal_separable_bands[n]
                a = x[j]
                X_dim_cen_j = chessboard_cenvalue1[j]
                OBJ_index_x = np.argmin(np.abs(X_dim_cen_j - a))
                x_OBJ_index_set.append(OBJ_index_x)

            x_OBJ_index_set = np.array(x_OBJ_index_set)
            sub = x_OBJ_index_set - target_OBJ_index_set
            x_target_index_differ_on_chess = np.linalg.norm(sub, ord=1)
            target_score = x_target_index_differ_on_chess / OBJ
            result[i] = np.exp(-target_score)

        result_2D = result.reshape(H, W)
        return result_2D[np.newaxis, np.newaxis, :, :]  # Reshape to (1, 1, H, W)