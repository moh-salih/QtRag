cmake_minimum_required(VERSION 3.16)
project(QtRag VERSION 1.0.0 LANGUAGES CXX)

find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(hnswlib REQUIRED)   

add_library(QtRag STATIC
    src/RagEngine.cpp
    src/RagTextChunker.cpp
    include/QtRag/QtRag.h
    include/QtRag/RagTypes.h
    include/QtRag/RagConfig.h
    include/QtRag/IRagStorage.h
    include/QtRag/IRagEmbedder.h
    include/QtRag/RagEngine.h
    include/QtRag/RagTextChunker.h
)
add_library(QtRag::QtRag ALIAS QtRag)

target_include_directories(QtRag PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(QtRag PUBLIC Qt6::Core hnswlib::hnswlib)
set_target_properties(QtRag PROPERTIES AUTOMOC ON CXX_STANDARD 23)

# --- Install (same pattern as QtLlama) ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS QtRag
    EXPORT QtRagTargets
    LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT QtRagTargets
    FILE      QtRagTargets.cmake
    NAMESPACE QtRag::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QtRag
)
configure_package_config_file(
    cmake/QtRagConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/QtRagConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QtRag
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/QtRagConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/QtRagConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/QtRagConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QtRag
)
