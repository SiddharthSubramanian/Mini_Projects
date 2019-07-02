library(gRain)

QMACR058 <- cptable(~QMACR058	  
              ,values = c(0.127, 0.8731)
              ,levels = c("No", "Yes"));


QM2184 <- cptable(~QM2184  
              ,values = c(0.25537, 0.74463)
              ,levels = c("No", "Yes"));

QM54 <- cptable(~QM54  
              ,values = c(0.2551, 0.7449)
              ,levels = c("No", "Yes"));


QM232 <- cptable(~QM232  
                ,values = c(0.24981, 0.75019)
                ,levels = c("No", "Yes"));
----------------------------------------------------------
	  

QM309 <- cptable(~QM309  
                 ,values = c(0.678457, 0.321543)
                 ,levels = c("No", "Yes"));

QM309_QM242 <- cptable(~QM242 | QM309
              ,values = c(0.994, 0.005,
                          1, 0)
              ,levels = c("no_tumour", "yes_tumour"));

QM309_QM224 <- cptable(~QM224 | QM309,
                       values=c(0.376466667,0.623533333,0.000140667,0.999859333),
                       levels = c("no_bloodinfection","yes_bloodinfection"))

QM309_QM297 <- cptable(~QM297 | QM309,
                       values=c(0.380133333,0.619866667,0.000140667,0.999859333),
                       levels = c("no_fammilySTD","yes_familySTD"))



QM309_QM90 <- cptable(~QM90 | QM309,
                       values=c(0.903133333,0.096866667,0.000422,0.999578),
                       levels = c("no_smoking","yes_smoking"))

----------------------------------------------------------------------------
  
  QM309_QM90_QM2130 <- cptable(~QM2130 | QM309 + QM90,
                        values=c(0.379936517,0.620063483,0.384721266,0.615278734,
                                 0.333333333,0.666666667,0,1),
                        levels = c("no_drugs","yes_drugs"))

QM309_QM90_QM2130_QM40 <- cptable(~QM40 | QM309 + QM90 + QM2130,
                             values=c(0.994171362,0.005828638,0,1,0.994633274,0.005366726,
                                      0,1,1,0,0,1,0,0,0,1),
                             levels = c("no_nervoussystem","yes_nervoussystem"))

  
----------------------------------------------------------------------------
QM309_QM224_QM217 <- cptable(~QM217 | QM309 + QM224
                         ,values = c(0.99, 0.0095626,1,0,
                                     1, 0,1,0)
                         ,levels = c("no_jaundice", "yes_jaundice"))

QM309_QM224_QM207 <- cptable(~QM207 | QM309 + QM224
                             ,values = c(0.984416504, 0.015583496,0.006201219,0.993798781,
                                         1, 0,0,1)
                             ,levels = c("no_stomach", "yes_stomach"))

QM309_QM224_QM249 <- cptable(~QM249 | QM309 + QM224
                             ,values = c(0.992562423, 0.007437577,0.006308136,0.993691864,
                                         1, 0,0,1)
                             ,levels = c("no_anameia", "yes_anameia"))

QM309_QM297_QM302 <- cptable(~QM302 | QM309 + QM297
                             ,values = c(0.993686426,0.006313574,0.0004302,0.9995698,
                                         1, 0,0,1)
                             ,levels = c("no_HIV", "yes_HIV"));


QM309_QM90_QM2130_QM40_QM194 <- cptable(~QM194 | QM309 + QM90 + QM2130 + QM40
                             ,values = c(0.970685949,0.029314051,1,0,0,0,1,0,
                                         0.964028777,0.035971223,1,0,0,0,1,0,
                                         1,0,0,0,0,0,1,0,
                                         0,0,0,0,0,0,1,0)
                             ,levels = c("no_heart",
                                         "yes_heart"))



#---------finalcalculation-------------------------------------------
  

underwriting.grain <- grain(compileCPT(list(QM309,QM309_QM242,QM309_QM224,QM309_QM297,QM309_QM90,QM309_QM90_QM2130,QM309_QM90_QM2130_QM40,QM309_QM224_QM217,QM309_QM224_QM207,
                                            QM309_QM224_QM249,QM309_QM297_QM302,QM309_QM90_QM2130_QM40_QM194)))
plot(underwriting.grain)
